# Environment Lock

This file records the environment snapshot used for the packaged numerical
artifacts and final smoke tests. The runnable dependency lock is the pinned
`requirements.txt`; this snapshot is included so a replication assistant can
quickly compare their local environment against the one used for packaging.

## Python Runtime

- Python: `3.11.9`

## Core Python Packages

The replication scripts import only the following scientific Python packages:

```text
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.8
scipy==1.17.1
```

These versions are pinned in both `requirements.txt` and `environment.yml`.

## TeX Runtime

- TeX engine used for the final package build: `pdflatex`
- TeX distribution check: `kpathsea version 6.4.2`
- The package includes `retention.bbl`, so the manuscript can be rebuilt with
  repeated `pdflatex` runs even when BibTeX is unavailable.

## Final Smoke Tests

The final package was checked with:

```bash
python3 -m py_compile *.py
bash -n run_reproduction.sh
./run_reproduction.sh --quick
pdflatex -interaction=nonstopmode -halt-on-error retention.tex
```

Quick mode writes into `quick_outputs/` and does not overwrite the published
CSV/PNG/PDF artifacts.
