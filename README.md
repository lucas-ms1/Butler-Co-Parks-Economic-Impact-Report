# Butler County Parks Economic Impact Report

This repository contains the final report and the curated analysis scripts
used to quantify housing and health impacts of parks in Butler County.

## Contents

- `final.pdf`: Final report for the project.
- `data_final/`: Clean, analysis-ready datasets used by the scripts.
- `scripts/`: Reproducible analysis pipeline and robustness checks.

## Reproducibility

1. Create an environment and install dependencies:
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
   - `pip install -r requirements.txt`

2. Run scripts in the logical order for each module:
   - Housing exposure + regressions: `scripts/02_parcel_distances_boundary_v2.py`,
     `scripts/add_tract_identifiers.py`, then `scripts/04_regression_models_clustered_boundary_v2.py`.
   - Health pipeline: `scripts/14_health_data_prep.py`,
     `scripts/15_tract_park_exposure.py`, `scripts/16_acs_dp03_prep.py`,
     `scripts/18_acs_dp02_education_prep.py`, `scripts/22_acs_dp05_population_prep.py`,
     `scripts/24_acs_dp05_demographics_prep.py`, `scripts/17_health_analysis_dataset.py`,
     then model scripts (e.g., `scripts/20_health_regression_models.py`,
     `scripts/23_park_gravity_index_models.py`).

## Notes

- The scripts assume local access to the underlying raw GIS/ACS inputs used in
  the project (not included here).
- Output files are written to `results/` and `figures/` when you run scripts.
