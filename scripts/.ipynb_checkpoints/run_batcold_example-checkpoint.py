from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xgenepy import FitObject, fit_edgepython, get_assignments_and_plot


def main() -> None:
    repo_root = REPO_ROOT
    xgener_root = repo_root.parent / "XgeneR"
    counts = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_counts.csv", index_col=0)
    metadata = pd.read_csv(xgener_root / "inst" / "extdata" / "BATcold_ballinger_metadata.csv", index_col=0)

    for trans_model in ("log_additive", "dominant"):
        fit_obj = FitObject(counts=counts, metadata=metadata, trans_model=trans_model)
        fit_obj = fit_edgepython(fit_obj)
        assignments = get_assignments_and_plot(fit_obj, make_plot=False)

        output_dir = repo_root / "outputs" / "batcold"
        output_dir.mkdir(parents=True, exist_ok=True)

        fit_obj.weights.to_csv(output_dir / f"{trans_model}_weights.csv")
        if fit_obj.tagwise_dispersion is not None:
            fit_obj.tagwise_dispersion.to_frame().to_csv(output_dir / f"{trans_model}_tagwise_dispersion.csv")
        fit_obj.raw_pvals.to_csv(output_dir / f"{trans_model}_raw_pvals.csv")
        fit_obj.bh_fdrs.to_csv(output_dir / f"{trans_model}_fdrs.csv")
        assignments.dataframe[["gene", "cis_prop"]].to_csv(
            output_dir / f"{trans_model}_proportion_cis.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
