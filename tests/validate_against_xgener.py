from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xgenepy import FitObject, fit_edgepython, get_assignments_and_plot


@dataclass
class Metric:
    scenario: str
    trans_model: str
    table_name: str
    rows: int
    cols: int
    mean_abs_diff: float
    max_abs_diff: float


def _compare_numeric_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    drop_columns: list[str] | None = None,
) -> tuple[int, int, float, float]:
    drop_columns = drop_columns or []

    a = actual.copy()
    e = expected.copy()

    for column in drop_columns:
        if column in a.columns:
            a = a.drop(columns=[column])
        if column in e.columns:
            e = e.drop(columns=[column])

    common_index = a.index.intersection(e.index)
    common_columns = [column for column in e.columns if column in a.columns]
    a = a.loc[common_index, common_columns]
    e = e.loc[common_index, common_columns]

    diff = np.abs(a.to_numpy(dtype=float) - e.to_numpy(dtype=float))
    return (
        len(common_index),
        len(common_columns),
        float(np.nanmean(diff)),
        float(np.nanmax(diff)),
    )


def _load_one_condition_inputs(xgener_root: Path, n_genes: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_counts.csv", index_col=0)
    metadata = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_metadata.csv", index_col=0)
    if n_genes is not None:
        counts = counts.iloc[:n_genes]
    return counts, metadata


def _load_multi_condition_inputs(xgener_root: Path, n_genes: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(xgener_root / "inst" / "extdata" / "ballinger_counts.csv", index_col=0)
    metadata = pd.read_csv(xgener_root / "inst" / "extdata" / "ballinger_metadata.csv", index_col=0)
    metadata.index = metadata["Sample"].astype(str)
    if n_genes is not None:
        counts = counts.iloc[:n_genes]
    return counts, metadata


def run_validation(n_genes: int | None) -> list[Metric]:
    root = PROJECT_ROOT.parent
    xgener_root = root / "XgeneR"
    fixture_root = xgener_root / "tests" / "fixtures"

    metrics: list[Metric] = []

    for trans_model in ("log_additive", "dominant"):
        # One-condition
        counts1, metadata1 = _load_one_condition_inputs(xgener_root, n_genes)
        fit1 = FitObject(counts=counts1, metadata=metadata1, trans_model=trans_model)
        fit1 = fit_edgepython(fit1)
        assign1 = get_assignments_and_plot(fit1, make_plot=False)

        expected_dir_one = fixture_root / "batcold"
        expected_weights1 = pd.read_csv(expected_dir_one / f"{trans_model}_weights.csv", index_col=0).iloc[: len(counts1)]
        expected_disp1 = pd.read_csv(expected_dir_one / f"{trans_model}_tagwise_dispersion.csv", index_col=0).iloc[: len(counts1)]
        expected_raw1 = pd.read_csv(expected_dir_one / f"{trans_model}_raw_pvals.csv", index_col=0).iloc[: len(counts1)]
        expected_fdr1 = pd.read_csv(expected_dir_one / f"{trans_model}_fdrs.csv", index_col=0).iloc[: len(counts1)]
        expected_prop1 = pd.read_csv(expected_dir_one / f"{trans_model}_proportion_cis.csv").iloc[: len(counts1)]

        actual_disp1 = fit1.tagwise_dispersion.to_frame()
        actual_prop1 = assign1.dataframe[["gene", "cis_prop"]].set_index("gene")
        expected_prop1 = expected_prop1.set_index("gene")

        for table_name, actual, expected, drops in [
            ("one_condition_weights", fit1.weights, expected_weights1, []),
            ("one_condition_tagwise_dispersion", actual_disp1, expected_disp1, []),
            ("one_condition_raw_pvals", fit1.raw_pvals, expected_raw1, ["Genes"]),
            ("one_condition_fdrs", fit1.bh_fdrs, expected_fdr1, ["Genes"]),
            ("one_condition_proportion_cis", actual_prop1, expected_prop1, []),
        ]:
            rows, cols, mean_abs, max_abs = _compare_numeric_frames(actual, expected, drop_columns=drops)
            metrics.append(Metric("one_condition", trans_model, table_name, rows, cols, mean_abs, max_abs))

        # Multi-condition
        counts2, metadata2 = _load_multi_condition_inputs(xgener_root, n_genes)
        fit2 = FitObject(
            counts=counts2,
            metadata=metadata2,
            trans_model=trans_model,
            fields_to_test=["Tissue", "Temperature"],
        )
        fit2 = fit_edgepython(fit2)

        expected_dir_multi = fixture_root / "multicondition"
        expected_design2 = pd.read_csv(expected_dir_multi / f"{trans_model}_design_matrix.csv", index_col=0).iloc[: len(counts2)]
        expected_weights2 = pd.read_csv(expected_dir_multi / f"{trans_model}_weights.csv", index_col=0).iloc[: len(counts2)]
        expected_disp2 = pd.read_csv(expected_dir_multi / f"{trans_model}_tagwise_dispersion.csv", index_col=0).iloc[: len(counts2)]
        expected_raw2 = pd.read_csv(expected_dir_multi / f"{trans_model}_raw_pvals.csv", index_col=0).iloc[: len(counts2)]
        expected_fdr2 = pd.read_csv(expected_dir_multi / f"{trans_model}_fdrs.csv", index_col=0).iloc[: len(counts2)]

        actual_disp2 = fit2.tagwise_dispersion.to_frame()

        for table_name, actual, expected, drops in [
            ("multi_condition_design_matrix", fit2.design_matrix_full, expected_design2, []),
            ("multi_condition_weights", fit2.weights, expected_weights2, []),
            ("multi_condition_tagwise_dispersion", actual_disp2, expected_disp2, []),
            ("multi_condition_raw_pvals", fit2.raw_pvals, expected_raw2, ["Genes"]),
            ("multi_condition_fdrs", fit2.bh_fdrs, expected_fdr2, ["Genes"]),
        ]:
            rows, cols, mean_abs, max_abs = _compare_numeric_frames(actual, expected, drop_columns=drops)
            metrics.append(Metric("multi_condition", trans_model, table_name, rows, cols, mean_abs, max_abs))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate XgenePy outputs against XgeneR fixtures.")
    parser.add_argument(
        "--n-genes",
        type=int,
        default=50,
        help="Number of leading genes to validate. Use -1 for full dataset.",
    )
    args = parser.parse_args()
    n_genes = None if args.n_genes == -1 else args.n_genes

    metrics = run_validation(n_genes=n_genes)
    df = pd.DataFrame([m.__dict__ for m in metrics])

    out_dir = PROJECT_ROOT / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xgenepy_vs_xgener_validation.csv"
    df.to_csv(out_path, index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))
    print(f"\nWrote validation metrics to: {out_path}")


if __name__ == "__main__":
    main()
