from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from xgenepy import FitObject, fit_edgepython, get_assignments_and_plot, get_fdrs


def _load_batcold_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    xgener_root = root.parent / "XgeneR"
    counts = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_counts.csv", index_col=0).iloc[:50]
    metadata = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_metadata.csv", index_col=0)
    return counts, metadata


def test_single_condition_design_matrix_matches_expected_columns() -> None:
    counts, metadata = _load_batcold_subset()
    fit_obj = FitObject(counts=counts, metadata=metadata, trans_model="log_additive")
    assert list(fit_obj.design_matrix_full.columns) == ["Intercept", "beta_cis", "beta_trans"]
    assert fit_obj.design_matrix_full.shape == (metadata.shape[0], 3)


def test_trans_models_use_centered_regulatory_weights() -> None:
    counts = pd.DataFrame(
        [[10, 12, 11, 13]],
        index=["gene1"],
        columns=["P1_sample", "P2_sample", "H1_sample", "H2_sample"],
    )
    metadata = pd.DataFrame(
        {"Allele": ["P1", "P2", "H1", "H2"]},
        index=counts.columns,
    )

    expected = {
        "log_additive": pd.DataFrame(
            {
                "Intercept": [1.0, 1.0, 1.0, 1.0],
                "beta_cis": [-0.5, 0.5, -0.5, 0.5],
                "beta_trans": [-0.5, 0.5, 0.0, 0.0],
            },
            index=counts.columns,
        ),
        "dominant": pd.DataFrame(
            {
                "Intercept": [1.0, 1.0, 1.0, 1.0],
                "beta_cis": [-0.5, 0.5, -0.5, 0.5],
                "beta_trans": [-0.5, 0.5, 0.5, 0.5],
            },
            index=counts.columns,
        ),
        "free": pd.DataFrame(
            {
                "Intercept": [1.0, 1.0, 1.0, 1.0],
                "beta_cis": [-0.5, 0.5, -0.5, 0.5],
                "beta_trans": [-0.5, 0.5, 0.0, 0.0],
                "beta_hybrid": [0.0, 0.0, 1.0, 1.0],
            },
            index=counts.columns,
        ),
    }

    for trans_model, expected_design in expected.items():
        fit_obj = FitObject(counts=counts, metadata=metadata, trans_model=trans_model)
        pd.testing.assert_frame_equal(fit_obj.design_matrix_full, expected_design)


def test_condition_interactions_use_centered_weights_times_indicators() -> None:
    counts = pd.DataFrame(
        [[10, 12, 11, 13, 14, 16, 15, 17]],
        index=["gene1"],
        columns=["P1_A", "P2_A", "H1_A", "H2_A", "P1_B", "P2_B", "H1_B", "H2_B"],
    )
    metadata = pd.DataFrame(
        {
            "Allele": ["P1", "P2", "H1", "H2"] * 2,
            "Condition": ["A"] * 4 + ["B"] * 4,
        },
        index=counts.columns,
    )
    fit_obj = FitObject(counts=counts, metadata=metadata, trans_model="free", fields_to_test=["Condition"])
    design = fit_obj.design_matrix_full
    indicator = (metadata["Condition"] == "B").astype(float).to_numpy()

    assert (design["beta_cis*condition-B"].to_numpy() == design["beta_cis"].to_numpy() * indicator).all()
    assert (design["beta_trans*condition-B"].to_numpy() == design["beta_trans"].to_numpy() * indicator).all()
    assert (design["beta_hybrid*condition-B"].to_numpy() == design["beta_hybrid"].to_numpy() * indicator).all()


def test_fdrs_match_statsmodels_bh() -> None:
    pvals = pd.Series([0.01, 0.04, 0.03, 0.2, 0.8])
    expected = multipletests(pvals, method="fdr_bh")[1]
    assert (get_fdrs(pvals) == expected).all()


def test_batcold_smoke_fit_runs_for_both_trans_models() -> None:
    counts, metadata = _load_batcold_subset()

    for trans_model in ("log_additive", "dominant", "free"):
        fit_obj = FitObject(counts=counts, metadata=metadata, trans_model=trans_model)
        fit_obj = fit_edgepython(fit_obj)
        assignments = get_assignments_and_plot(fit_obj, make_plot=False)

        assert fit_obj.weights is not None
        assert fit_obj.raw_pvals is not None
        assert fit_obj.bh_fdrs is not None
        assert fit_obj.tagwise_dispersion is not None
        assert fit_obj.weights.shape[0] == counts.shape[0]
        assert "null: no cis" in fit_obj.raw_pvals.columns
        assert "null: no trans" in fit_obj.raw_pvals.columns
        assert "cis_prop" in assignments.dataframe.columns
        assert len(assignments.dataframe) == counts.shape[0]
