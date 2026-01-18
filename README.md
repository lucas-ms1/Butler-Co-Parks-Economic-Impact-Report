# Butler County Parks Economic Impact Report

This repository contains the final report and the curated analysis scripts
used to quantify housing and health impacts of parks in Butler County.

## Contents

- `final.pdf`: Final report for the project.
- `data_final/`: Clean, analysis-ready datasets used by the scripts.
- `scripts/`: Reproducible analysis pipeline and robustness checks.

## Results Highlights (See `final.pdf` for full tables/figures)

- Housing capitalization is estimated with boundary-based park exposure, tract-clustered
  standard errors, and fixed-effect benchmarks to separate local vs. bundled effects.
- Health outcomes are analyzed at the tract level using park proximity and a
  continuous Park Gravity Index (PGI), with socioeconomic controls and population weighting.
- Robustness includes influence diagnostics, spatial autocorrelation tests, and
  sensitivity to the PGI distance-decay parameter.
- Economic valuation translates model effects into property value uplift and
  health cost savings scenarios.

## Data Structure

- `final.pdf`: Final written report.
- `data_final/`
  - `housing_regression_ready.csv` / `housing_regression_ready_with_tract.csv`
  - `housing_regression_ready.xlsx`
  - `butler_tract_health_model_data.csv`
  - `butler_tract_health_model_data_with_greenness.csv`
  - `*.gpkg` tract-level GeoPackages used for spatial modeling
- `scripts/`
  - Housing: exposure construction, clustered regressions, distance-band benefits
  - Health: tract dataset prep, PGI models, robustness diagnostics
  - Phase 2: quality-weighted analysis, valuation, visualizations

## Data Sources Needed (Not Included Here)

The scripts expect local access to the following raw inputs:

- Parcel polygons: `data_raw/CURRENTPARCELS/CURRENTPARCELS.shp`
- Park polygons (NDVI/OSM export): `data_raw/ButlerParks_NDVI_mean_S2_2025_JunAug.csv`
- Road network (for network distances): `data_raw/Roads/Roads.shp`
- CDC PLACES (tract health outcomes): `data_raw/Health and Census Tracts/PLACES__Census_Tract_Data_*.csv`
- TIGER/Line tracts: `data_raw/Health and Census Tracts/tl_2025_39_tract.zip`
- ACS DP03 (economics): `data_raw/Health and Census Tracts/ACSDP5Y2023.DP03_*.zip`
- ACS DP02 (education): `data_raw/Health and Census Tracts/ACSDP5Y2023.DP02_*.zip`
- ACS DP05 (population/demographics): `data_raw/Health and Census Tracts/ACSDP5Y2023.DP05_*.zip`
- Park amenities (Phase 2 quality weighting): `data_raw/park_amenities.csv`
- Optional quality metrics: `data_processed/tract_park_quality_metrics.csv`

## How I Handled the Data

- Standardized CRS across spatial layers and used Ohio State Plane South for
  distance calculations.
- Filtered residential parcels, computed Euclidean and network distances to park
  boundaries, and created consistent distance bands.
- Attached census tract GEOIDs to parcel sales for clustered inference.
- Built tract-level health datasets by merging CDC PLACES outcomes with ACS
  economic, education, population, and demographic covariates.
- Created derived variables (log prices, fixed effects, PGI z-scores, density
  controls) and used HC1 or cluster-robust SEs plus WLS (population-weighted)
  specifications where appropriate.

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

- Output files are written to `results/` and `figures/` when you run scripts.
