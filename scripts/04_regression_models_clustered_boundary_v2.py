"""
V2 housing hedonic regressions using boundary-based park exposure.

Primary (headline): MetroParks boundary distances.
Robustness (appendix): All-parks boundary distances.

Inputs:
- data_final/housing_regression_ready_with_tract.csv
- data_intermediate/v2/parcel_boundary_distances_v2.csv (PIN-level boundary distances)

Outputs:
- results/v2/housing_regression_clustered_boundary_v2.csv
- results/v2/housing_regression_clustered_boundary_v2_summary.txt
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def collapse_band_0to3(band: Any) -> str:
    b = str(band)
    keep = {"0-0.1", "0.1-0.25", "0.25-0.75", "0.75-1.5", "1.5-3"}
    return b if b in keep else "1.5-3"


def fit_clustered(formula: str, data: pd.DataFrame, cluster_var: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    data_use = data.dropna(subset=[cluster_var]).copy()
    y, X = patsy.dmatrices(formula, data=data_use, return_type="dataframe")
    groups = data_use.loc[y.index, cluster_var]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def extract_rows(
    res: sm.regression.linear_model.RegressionResultsWrapper, model_name: str, cluster_var: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = res.params
    ci = res.conf_int()

    def add(var: str, pretty: str) -> None:
        coef = float(params[var])
        se = float(res.bse[var])
        p = float(res.pvalues[var])
        ci_low, ci_high = float(ci.loc[var, 0]), float(ci.loc[var, 1])

        pct = (np.exp(coef) - 1.0) * 100.0
        pct_low = (np.exp(ci_low) - 1.0) * 100.0
        pct_high = (np.exp(ci_high) - 1.0) * 100.0

        rows.append(
            {
                "model": model_name,
                "cluster_var": cluster_var,
                "variable": pretty,
                "coef": coef,
                "std_error": se,
                "pvalue": p,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "pct_premium": pct,
                "pct_premium_lower": pct_low,
                "pct_premium_upper": pct_high,
                "n_obs": int(res.nobs),
                "r_squared": float(res.rsquared),
            }
        )

    for var in params.index:
        if var.startswith("NearPark_Boundary_"):
            add(var, var)
        if var.startswith("C(DistBand_Boundary_"):
            m = re.search(r"\[T\.(.+?)\]$", var)
            band = m.group(1) if m else var
            add(var, f"{var.split('[')[0]}:{band}")

    return rows


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    sales_path = os.path.join(project_root, "data_final", "housing_regression_ready_with_tract.csv")
    boundary_path = os.path.join(project_root, "data_intermediate", "v2", "parcel_boundary_distances_v2.csv")

    output_dir = os.path.join(project_root, "results", "v2")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "housing_regression_clustered_boundary_v2.csv")
    output_txt = os.path.join(output_dir, "housing_regression_clustered_boundary_v2_summary.txt")

    print("=" * 80)
    print("V2 HOUSING REGRESSION (BOUNDARY EXPOSURE, CLUSTERED SE)")
    print("=" * 80)

    if not os.path.exists(boundary_path):
        raise FileNotFoundError(
            "Expected boundary distance file not found. Run scripts/02_parcel_distances_boundary_v2.py first:\n"
            f"  {boundary_path}"
        )

    df = pd.read_csv(sales_path, low_memory=False)
    df["PIN"] = df["PIN"].astype(str)
    boundary = pd.read_csv(boundary_path, low_memory=False)
    boundary["PIN"] = boundary["PIN"].astype(str)

    # Merge boundary exposure
    df = df.merge(boundary, on="PIN", how="left")
    if "dist_to_park_boundary_miles_metro" not in df.columns:
        raise ValueError("Boundary distance merge failed (missing dist_to_park_boundary_miles_metro).")

    # Core vars
    df["ln_price"] = np.log(df["PRICE"])
    df["SALEDT"] = pd.to_datetime(df["SALEDT"])
    df["SaleYear"] = df["SALEDT"].dt.year

    # Distance bands (0–3 focus)
    df["DistBand_Boundary_Metro_0to3"] = df["DistBand_Boundary_Metro"].apply(collapse_band_0to3)
    df["DistBand_Boundary_All_0to3"] = df["DistBand_Boundary_All"].apply(collapse_band_0to3)

    if "GEOID" not in df.columns or df["GEOID"].isna().all():
        raise ValueError("GEOID missing; run scripts/add_tract_identifiers.py to create the input file with GEOID.")

    reference_band = "1.5-3"
    cluster_var = "GEOID"

    formulas = {
        # MetroParks (headline)
        "Metro_Binary_Bundled": "ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear)",
        "Metro_Binary_WithSchoolDistFE": "ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)",
        "Metro_DistanceBands_Bundled": f'ln_price ~ C(DistBand_Boundary_Metro_0to3, Treatment(reference="{reference_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear)',
        "Metro_DistanceBands_WithSchoolDistFE": f'ln_price ~ C(DistBand_Boundary_Metro_0to3, Treatment(reference="{reference_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)',
        # All parks (robustness)
        "AllParks_Binary_Bundled": "ln_price ~ NearPark_Boundary_All + SQFT + YEAR_BUILT + ACRES + C(SaleYear)",
        "AllParks_Binary_WithSchoolDistFE": "ln_price ~ NearPark_Boundary_All + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)",
        "AllParks_DistanceBands_Bundled": f'ln_price ~ C(DistBand_Boundary_All_0to3, Treatment(reference="{reference_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear)',
        "AllParks_DistanceBands_WithSchoolDistFE": f'ln_price ~ C(DistBand_Boundary_All_0to3, Treatment(reference="{reference_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)',
    }

    rows: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    summary_lines.append("V2 HOUSING REGRESSION (BOUNDARY EXPOSURE, CLUSTERED SE @ TRACT)")
    summary_lines.append(f"Input sales: {os.path.relpath(sales_path, project_root)}")
    summary_lines.append(f"Boundary distances: {os.path.relpath(boundary_path, project_root)}")
    summary_lines.append("")

    for name, formula in formulas.items():
        res = fit_clustered(formula, df, cluster_var=cluster_var)
        rows.extend(extract_rows(res, model_name=name, cluster_var=cluster_var))
        key_var = "NearPark_Boundary_Metro" if "Metro_Binary" in name else "NearPark_Boundary_All" if "AllParks_Binary" in name else None
        if key_var and key_var in res.params.index:
            b = float(res.params[key_var])
            se = float(res.bse[key_var])
            pct = (np.exp(b) - 1.0) * 100.0
            summary_lines.append(f"{name}: {key_var}={b:.6f} (SE={se:.6f}) => {pct:.2f}%")

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    with open(output_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nWrote: {output_csv}")
    print(f"Wrote: {output_txt}")


if __name__ == "__main__":
    main()
