from pathlib import Path

import pandas as pd

from xgenepy import FitObject, fit_edgepython, get_assignments_and_plot


def _load_batcold_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    xgener_root = root.parent / "XgeneR"
    counts = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_counts.csv", index_col=0)
    metadata = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_metadata.csv", index_col=0)
    return counts, metadata


def test_single_condition_design_matrix_matches_expected_columns() -> None:
    counts, metadata = _load_batcold_subset()
    fit_obj = FitObject(counts=counts, metadata=metadata, trans_model="log_additive")
    assert list(fit_obj.design_matrix_full.columns) == ["Intercept", "beta_cis", "beta_trans"]
    assert fit_obj.design_matrix_full.shape == (metadata.shape[0], 3)


def test_batcold_smoke_fit_runs_for_both_trans_models() -> None:
    counts, metadata = _load_batcold_subset()

    for trans_model in ("log_additive", "dominant"):
        fit_obj = FitObject(counts=counts, metadata=metadata, trans_model=trans_model)
        fit_obj = fit_edgepython(fit_obj)
        assignments = get_assignments_and_plot(fit_obj, make_plot=False)

        assert fit_obj.weights is not None
        assert fit_obj.raw_pvals is not None
        assert fit_obj.bh_fdrs is not None
        assert fit_obj.tagwise_dispersion is not None
        assert fit_obj.weights.shape[0] == counts.shape[0]
        assert "beta_cis" in fit_obj.raw_pvals.columns
        assert "beta_trans" in fit_obj.raw_pvals.columns
        assert "cis_prop" in assignments.dataframe.columns
        assert len(assignments.dataframe) == counts.shape[0]
