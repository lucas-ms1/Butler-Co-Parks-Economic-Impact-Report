"""
V2 supervisor-facing health redesign: continuous Park Gravity Index (PGI) regressions.

Outputs:
- results/v2/health_pgi_regression_v2.csv
- figures/v2/health_pgi_coefficients_v2.png
"""

from __future__ import annotations

import os
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist


def compute_pgi_z(tracts_gdf: gpd.GeoDataFrame, parks_gdf: gpd.GeoDataFrame, lambda_d: float = -1.76) -> pd.Series:
    """
    Compute Macfarlane et al. (2020) Park Gravity Index and z-score normalize.
    PGI_i = ln( sum_j d_ij^lambda )
    where d_ij is Euclidean distance in meters from tract centroid i to park j.
    """
    if tracts_gdf.crs is None:
        raise ValueError("Tracts GeoDataFrame must have a CRS.")
    if parks_gdf.crs is None:
        raise ValueError("Parks GeoDataFrame must have a CRS.")
    if tracts_gdf.crs != parks_gdf.crs:
        parks_gdf = parks_gdf.to_crs(tracts_gdf.crs)

    centroids = tracts_gdf.geometry.centroid
    centroids_xy = np.array([[geom.x, geom.y] for geom in centroids])

    if parks_gdf.geometry.geom_type.iloc[0] == "Point":
        parks_xy = np.array([[geom.x, geom.y] for geom in parks_gdf.geometry])
    else:
        parks_xy = np.array([[geom.x, geom.y] for geom in parks_gdf.geometry.centroid])

    dist_ft = cdist(centroids_xy, parks_xy, metric="euclidean")
    US_SURVEY_FT_TO_M = 0.3048006096
    dist_m = dist_ft * US_SURVEY_FT_TO_M
    dist_m = np.maximum(dist_m, 1.0)

    gravity_components = np.exp(lambda_d * np.log(dist_m))
    gravity_sum = np.sum(gravity_components, axis=1)
    pgi = np.log(gravity_sum)
    pgi_z = (pgi - pgi.mean()) / pgi.std(ddof=0)

    return pd.Series(pgi_z, index=tracts_gdf.index, name="park_gravity_index_z")


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    weight_col: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    cols = [outcome] + covariates
    if weight_col:
        cols.append(weight_col)

    d = df.dropna(subset=cols).copy()
    y = d[outcome]
    X = sm.add_constant(d[covariates], has_constant="add")

    if weight_col:
        w = d[weight_col]
        res = sm.WLS(y, X, weights=w).fit(cov_type="HC1")
    else:
        res = sm.OLS(y, X).fit(cov_type="HC1")
    return res


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    tracts_path = os.path.join(project_root, "data_final", "butler_tract_health_model_data_with_greenness.gpkg")
    parks_path = os.path.join(project_root, "data_intermediate", "butler_county_parks.shp")

    output_dir = os.path.join(project_root, "results", "v2")
    fig_dir = os.path.join(project_root, "figures", "v2")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "health_pgi_regression_v2.csv")
    output_fig = os.path.join(fig_dir, "health_pgi_coefficients_v2.png")

    print("=" * 80)
    print("V2 HEALTH PGI REGRESSIONS (SUPERVISOR-FACING)")
    print("=" * 80)
    print(f"Loading tracts: {tracts_path}")
    gdf = gpd.read_file(tracts_path)
    print(f"  Tracts: {len(gdf):,} | CRS: {gdf.crs}")

    print(f"Loading parks: {parks_path}")
    parks = gpd.read_file(parks_path)
    print(f"  Parks: {len(parks):,} | CRS: {parks.crs}")

    target_crs = "EPSG:3402"
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    if str(parks.crs) != target_crs:
        parks = parks.to_crs(target_crs)

    gdf["park_gravity_index_z"] = compute_pgi_z(gdf, parks, lambda_d=-1.76)

    df = gdf.drop(columns=["geometry"]).copy()

    outcomes = [
        ("MHLTH_CrudePrev", "Frequent Mental Distress"),
        ("OBESITY_CrudePrev", "Obesity"),
        ("LPA_CrudePrev", "Physical Inactivity"),
    ]

    covariates = [
        "park_gravity_index_z",
        "median_household_income",
        "pct_families_below_poverty",
        "unemployment_rate",
        "pct_bachelors_degree_or_higher",
        "pct_under_18",
        "pct_65_and_over",
        "pct_black",
        "pct_hispanic",
    ]

    if "ALAND" in df.columns:
        SQM_TO_SQMI = 3.861021585e-7
        area_sqmi = df["ALAND"] * SQM_TO_SQMI
        df["pop_density_sqmi"] = df["total_population"] / area_sqmi.replace(0, np.nan)
        df["log_pop_density"] = np.log(df["pop_density_sqmi"].replace(0, np.nan))
        covariates.append("log_pop_density")

    results: list[dict[str, Any]] = []

    for outcome_col, label in outcomes:
        if outcome_col not in df.columns:
            print(f"Skipping {label} (missing column {outcome_col})")
            continue

        res_ols = fit_model(df, outcome_col, covariates=covariates, weight_col=None)
        b = float(res_ols.params["park_gravity_index_z"])
        se = float(res_ols.bse["park_gravity_index_z"])
        p = float(res_ols.pvalues["park_gravity_index_z"])
        ci_low, ci_high = res_ols.conf_int().loc["park_gravity_index_z"].tolist()
        results.append(
            {
                "outcome": label,
                "outcome_col": outcome_col,
                "model_type": "OLS_HC1",
                "beta_pp_per_1SD_PGI": b,
                "std_error": se,
                "pvalue": p,
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
                "n_obs": int(res_ols.nobs),
                "r_squared": float(res_ols.rsquared),
                "sign_convention": "beta>0 => higher prevalence (worse)",
            }
        )

        res_wls = fit_model(df, outcome_col, covariates=covariates, weight_col="total_population")
        b = float(res_wls.params["park_gravity_index_z"])
        se = float(res_wls.bse["park_gravity_index_z"])
        p = float(res_wls.pvalues["park_gravity_index_z"])
        ci_low, ci_high = res_wls.conf_int().loc["park_gravity_index_z"].tolist()
        results.append(
            {
                "outcome": label,
                "outcome_col": outcome_col,
                "model_type": "WLS_pop_HC1",
                "beta_pp_per_1SD_PGI": b,
                "std_error": se,
                "pvalue": p,
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
                "n_obs": int(res_wls.nobs),
                "r_squared": float(res_wls.rsquared),
                "sign_convention": "beta>0 => higher prevalence (worse)",
            }
        )

        print(f"{label}: WLS beta={b:.3f}pp per +1SD PGI (p={p:.4g})")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nWrote: {output_csv}")

    wls = out_df[out_df["model_type"] == "WLS_pop_HC1"].copy()
    if len(wls) > 0:
        wls = wls.sort_values("beta_pp_per_1SD_PGI")
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(wls))
        x = wls["beta_pp_per_1SD_PGI"].values
        xerr = np.vstack([x - wls["ci_lower"].values, wls["ci_upper"].values - x])

        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=3)
        ax.axvline(0, color="#555", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(wls["outcome"].values)
        ax.set_xlabel("β (pp prevalence per +1 SD PGI) [WLS, HC1]")
        ax.set_title("Health Associations vs Park Gravity Index (PGI)\nPositive β = higher prevalence (worse)")
        fig.tight_layout()
        fig.savefig(output_fig, dpi=200)
        plt.close(fig)
        print(f"Wrote: {output_fig}")


if __name__ == "__main__":
    main()
