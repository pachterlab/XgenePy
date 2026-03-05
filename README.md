# XgenePy

`XgenePy` is a Python port of the `XgeneR` package for modeling cis and trans
regulatory differences between homozygous strains. It mirrors the core `XgeneR`
workflow while replacing the R `edgeR` dependency with the Python
`edgepython` package.

The package centers on:

- building the same allele-aware design matrices used by `XgeneR`
- fitting negative binomial GLMs with `edgepython`
- testing cis and trans hypotheses
- generating the same assignment and proportion-cis summaries
- producing diagnostic plots in matplotlib

## Install

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate xgenepy
```

Then install the package in editable mode:

```bash
pip install -e .
```

## Example

```python
import pandas as pd
from xgenepy import FitObject, fit_edgepython, get_assignments_and_plot

counts = pd.read_csv("../XgeneR/inst/extdata/BATcold_ballinger_counts.csv", index_col=0)
metadata = pd.read_csv("../XgeneR/inst/extdata/BATcold_ballinger_metadata.csv", index_col=0)

fit_obj = FitObject(counts=counts, metadata=metadata, trans_model="log_additive")
fit_obj = fit_edgepython(fit_obj)

results = get_assignments_and_plot(fit_obj, make_plot=False)
print(results.dataframe.head())
```

## Project Layout

- `src/xgenepy/model.py`: model construction, design matrices, contrasts, and fitting
- `src/xgenepy/plotting.py`: plotting and assignment helpers
- `data/`: copied example datasets from `XgeneR`, including `BATcold`, `cold`, and full Ballinger files
- `notebooks/`: Jupyter notebook equivalents of the `XgeneR` vignettes
- `tests/`: pytest coverage for design and fitting smoke tests
- `scripts/`: example script for running the BATcold dataset
