from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from edgepython import calc_norm_factors, estimate_disp, glm_fit, glm_lrt, make_dgelist


INTERACTION_DESIGNATOR = "*"


def _as_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    return list(values)


def _flatten_unique_categories(metadata: pd.DataFrame, fields_to_test: list[str]) -> dict[str, list[str]]:
    return {field_name: [str(v) for v in pd.unique(metadata[field_name])] for field_name in fields_to_test}


def make_field_level_label(field: str, level: str) -> str:
    return f"{field.lower()}-{level}"


def _validate_metadata(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if counts.columns.empty:
        raise ValueError("Counts matrix must have sample IDs as column names.")
    if counts.index.empty:
        raise ValueError("Counts matrix must have gene IDs as row names.")
    if "Allele" not in metadata.columns:
        raise ValueError("Metadata must include an 'Allele' column.")
    if not counts.columns.isin(metadata.index).all():
        raise ValueError("All count matrix sample IDs must exist in the metadata index.")
    return metadata.loc[counts.columns]


def convert_metadata(metadata: pd.DataFrame, ref: dict[str, str] | None = None) -> pd.DataFrame:
    if "Allele" not in metadata.columns:
        raise ValueError("Metadata must include an 'Allele' column.")

    ref = ref or {}
    metadata_reformatted = pd.DataFrame(index=metadata.index)
    metadata_reformatted["Allele"] = metadata["Allele"].astype(str)

    for cond in [column for column in metadata.columns if column != "Allele"]:
        categories = [str(v) for v in pd.unique(metadata[cond])]
        reference = ref.get(cond, categories[0])
        for category in categories:
            if category == reference:
                continue
            encoded_name = make_field_level_label(cond, category)
            metadata_reformatted[encoded_name] = (metadata[cond].astype(str) == category).astype(int)

    return metadata_reformatted


def get_design_vector(
    metadata: pd.DataFrame,
    covariate_cols: list[str] | None = None,
    fields_to_test: list[str] | None = None,
    ref: dict[str, str] | None = None,
    higher_order_interactions: list[str] | None = None,
    interaction_designator: str = INTERACTION_DESIGNATOR,
) -> tuple[list[str], dict[str, str] | None]:
    del covariate_cols

    design = ["Reg"]
    ref_categories: dict[str, str] | None = None
    fields = _as_list(fields_to_test)
    ref = ref or {}

    if fields:
        ref_categories = {}
        for field_name in fields:
            categories = [str(v) for v in pd.unique(metadata[field_name])]
            reference = ref.get(field_name, categories[0])
            ref_categories[field_name] = reference
            for category in categories:
                if category == reference:
                    continue
                level_label = make_field_level_label(field_name, category)
                design.append(f"Reg{interaction_designator}{level_label}")
                design.append(level_label)

    for term in _as_list(higher_order_interactions):
        design.append(term)

    return design, ref_categories


def _build_design_matrix(
    metadata_reformatted: pd.DataFrame,
    covariate_cols: list[str] | None,
    fields_to_test: list[str] | None,
    design: list[str],
    trans_model: str,
    reg_designator: str = "Reg",
    interact_designator: str = INTERACTION_DESIGNATOR,
) -> pd.DataFrame:
    if not design or design[0] != reg_designator:
        raise ValueError("Design must start with the Reg designator.")
    if "Allele" not in metadata_reformatted.columns:
        raise ValueError("Reformatted metadata must include an 'Allele' column.")

    allele = metadata_reformatted["Allele"]
    p1 = allele.eq("P1").to_numpy()
    p2 = allele.eq("P2").to_numpy()
    h1 = allele.eq("H1").to_numpy()
    h2 = allele.eq("H2").to_numpy()

    n_samples = len(metadata_reformatted)
    design_columns: list[np.ndarray] = [np.ones(n_samples, dtype=float)]
    weight_names = ["Intercept"]

    beta_cis = (p2 | h2).astype(float)
    if trans_model == "log_additive":
        beta_trans = np.where(p1, 1.0, np.where(h1 | h2, 0.5, 0.0))
    elif trans_model == "dominant":
        beta_trans = (p1 | h1 | h2).astype(float)
    else:
        raise ValueError("trans_model must be either 'log_additive' or 'dominant'.")

    design_columns.extend([beta_cis, beta_trans])
    weight_names.extend(["beta_cis", "beta_trans"])

    if not _as_list(fields_to_test) and _as_list(covariate_cols):
        covariate_matrix = metadata_reformatted.drop(columns=["Allele"])
        for column_name in covariate_matrix.columns:
            design_columns.append(covariate_matrix[column_name].to_numpy(dtype=float))
            weight_names.append(column_name)

    if _as_list(fields_to_test):
        for des in design[1:]:
            if des.startswith(f"{reg_designator}{interact_designator}"):
                conds = des.split(interact_designator)[1:]
                cond_filt = np.ones(n_samples, dtype=bool)
                for cond in conds:
                    cond_filt &= metadata_reformatted[cond].to_numpy(dtype=bool)
                name = interact_designator.join(conds)
                beta_cis_cond = ((p2 | h2) & cond_filt).astype(float)
                if trans_model == "log_additive":
                    beta_trans_cond = np.where(
                        p1 & cond_filt,
                        1.0,
                        np.where((h1 | h2) & cond_filt, 0.5, 0.0),
                    )
                else:
                    beta_trans_cond = ((p1 | h1 | h2) & cond_filt).astype(float)

                design_columns.extend([beta_cis_cond, beta_trans_cond])
                weight_names.extend([f"beta_cis*{name}", f"beta_trans*{name}"])
            else:
                conds = des.split(interact_designator)
                cond_filt = np.ones(n_samples, dtype=bool)
                for cond in conds:
                    cond_filt &= metadata_reformatted[cond].to_numpy(dtype=bool)
                name = interact_designator.join(conds)
                design_columns.append(cond_filt.astype(float))
                weight_names.append(f"beta_{name}")

    matrix = np.column_stack(design_columns)
    return pd.DataFrame(matrix, index=metadata_reformatted.index, columns=weight_names)


def get_fdrs(pvals: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    pvals_arr = np.asarray(pvals, dtype=float)
    num_test = len(pvals_arr)
    order_idx = np.argsort(pvals_arr)
    sorted_p = pvals_arr[order_idx]
    fdr_sorted = (np.arange(1, num_test + 1, dtype=float) / float(num_test)) * sorted_p
    fdr_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]
    fdr = np.empty(num_test, dtype=float)
    fdr[order_idx] = fdr_sorted
    return fdr


@dataclass
class FitObject:
    counts: pd.DataFrame
    metadata: pd.DataFrame
    trans_model: str = "log_additive"
    covariate_cols: list[str] | None = None
    fields_to_test: list[str] | None = None
    ref: dict[str, str] | None = None
    higher_order_interactions: list[str] | None = None
    design: list[str] = field(init=False)
    metadata_reformatted: pd.DataFrame = field(init=False)
    design_matrix_full: pd.DataFrame = field(init=False)
    raw_pvals: pd.DataFrame | None = field(default=None, init=False)
    bh_fdrs: pd.DataFrame | None = field(default=None, init=False)
    weights: pd.DataFrame | None = field(default=None, init=False)
    tagwise_dispersion: pd.Series | None = field(default=None, init=False)
    dge: Any = field(default=None, init=False)
    fit_result: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.counts, pd.DataFrame):
            self.counts = pd.DataFrame(self.counts)
        if not isinstance(self.metadata, pd.DataFrame):
            self.metadata = pd.DataFrame(self.metadata)

        self.counts = self.counts.copy()
        self.metadata = self.metadata.copy()

        self.counts.columns = self.counts.columns.map(str)
        self.counts.index = self.counts.index.map(str)
        self.metadata.index = self.metadata.index.map(str)

        self.metadata = _validate_metadata(self.counts, self.metadata)

        cols_to_include = ["Allele"]
        cols_to_include.extend(_as_list(self.fields_to_test))
        cols_to_include.extend(_as_list(self.covariate_cols))
        cols_to_include = [column for column in cols_to_include if column in self.metadata.columns]

        self.metadata = self.metadata.loc[:, cols_to_include]
        self.design, self.ref = get_design_vector(
            self.metadata,
            covariate_cols=self.covariate_cols,
            fields_to_test=self.fields_to_test,
            ref=self.ref,
            higher_order_interactions=self.higher_order_interactions,
        )
        self.metadata_reformatted = convert_metadata(self.metadata, self.ref)
        self.design_matrix_full = _build_design_matrix(
            self.metadata_reformatted,
            covariate_cols=self.covariate_cols,
            fields_to_test=self.fields_to_test,
            design=self.design,
            trans_model=self.trans_model,
        )


def _normalize_coef_indices(indices: np.ndarray, n_coef: int) -> list[int]:
    if indices.size == 0:
        raise ValueError("Coefficient index vector cannot be empty.")
    if np.any(indices < 0):
        raise ValueError("Coefficient indices must be non-negative.")
    if np.any(indices == 0):
        if np.any(indices >= n_coef):
            raise ValueError(f"0-based coefficient indices must be in [0, {n_coef - 1}].")
        return indices.astype(int).tolist()
    if np.any(indices > n_coef):
        raise ValueError(f"1-based coefficient indices must be in [1, {n_coef}].")
    return (indices.astype(int) - 1).tolist()


def fit_edgepython(object: FitObject, test: dict[str, np.ndarray | list[float] | list[int]] | None = None) -> FitObject:
    gene_names = object.counts.index.to_list()
    dge = make_dgelist(object.counts)
    dge = calc_norm_factors(dge)
    dge = estimate_disp(dge, design=None)
    fit = glm_fit(dge, design=object.design_matrix_full.to_numpy(dtype=float), dispersion=dge.get("tagwise.dispersion"))

    weight_names = object.design_matrix_full.columns.to_list()
    n_coef = len(weight_names)
    cis_coef_idx = np.array([i for i, name in enumerate(weight_names) if "cis" in name], dtype=int)
    trans_coef_idx = np.array([i for i, name in enumerate(weight_names) if "trans" in name], dtype=int)
    if cis_coef_idx.size == 0:
        raise ValueError("No coefficient names contain 'cis'; cannot run default cis test.")
    if trans_coef_idx.size == 0:
        raise ValueError("No coefficient names contain 'trans'; cannot run default trans test.")

    if test is not None:
        if not isinstance(test, dict) or any(not key for key in test):
            raise ValueError("`test` must be a dict with non-empty null-name keys.")

    raw_pval_dict: dict[str, Any] = {"Genes": gene_names}
    fdr_dict: dict[str, Any] = {"Genes": gene_names}

    default_tests: list[tuple[str, np.ndarray]] = [
        ("null: no cis", cis_coef_idx),
        ("null: no trans", trans_coef_idx),
    ]
    for test_name, coef_idx in default_tests:
        lrt = glm_lrt(fit, coef=coef_idx.tolist())
        pvals = lrt["table"]["PValue"].to_numpy(dtype=float)
        raw_pval_dict[test_name] = pvals
        fdr_dict[test_name] = get_fdrs(pvals)

    if _as_list(object.fields_to_test):
        for field_name in _as_list(object.fields_to_test):
            field_tag = field_name.lower()
            cis_prefix = f"beta_cis*{field_tag}-"
            trans_prefix = f"beta_trans*{field_tag}-"
            cis_field_idx = np.array([i for i, name in enumerate(weight_names) if name.startswith(cis_prefix)], dtype=int)
            trans_field_idx = np.array([i for i, name in enumerate(weight_names) if name.startswith(trans_prefix)], dtype=int)

            if cis_field_idx.size > 0:
                lrt = glm_lrt(fit, coef=cis_field_idx.tolist())
                pvals = lrt["table"]["PValue"].to_numpy(dtype=float)
                key = f"null: no {field_name} cis"
                raw_pval_dict[key] = pvals
                fdr_dict[key] = get_fdrs(pvals)
            if trans_field_idx.size > 0:
                lrt = glm_lrt(fit, coef=trans_field_idx.tolist())
                pvals = lrt["table"]["PValue"].to_numpy(dtype=float)
                key = f"null: no {field_name} trans"
                raw_pval_dict[key] = pvals
                fdr_dict[key] = get_fdrs(pvals)

    if test is not None:
        for test_name, test_vector in test.items():
            vector = np.asarray(test_vector, dtype=float)
            if vector.ndim != 1:
                raise ValueError(f"Custom test `{test_name}` must be a 1D numeric vector.")

            if vector.size == n_coef:
                lrt = glm_lrt(fit, contrast=vector.astype(float))
            else:
                coef_idx = _normalize_coef_indices(vector.astype(int), n_coef)
                lrt = glm_lrt(fit, coef=coef_idx)

            pvals = lrt["table"]["PValue"].to_numpy(dtype=float)
            raw_pval_dict[test_name] = pvals
            fdr_dict[test_name] = get_fdrs(pvals)

    object.raw_pvals = pd.DataFrame(raw_pval_dict, index=gene_names)
    object.bh_fdrs = pd.DataFrame(fdr_dict, index=gene_names)
    object.weights = pd.DataFrame(
        np.asarray(fit["coefficients"], dtype=float),
        index=gene_names,
        columns=object.design_matrix_full.columns,
    )

    dispersion_values = dge.get("tagwise.dispersion")
    if dispersion_values is not None:
        object.tagwise_dispersion = pd.Series(np.asarray(dispersion_values, dtype=float), index=gene_names, name="tagwise_dispersion")

    object.dge = dge
    object.fit_result = fit
    return object
