"""
V2 housing neighborhood FE benchmarks using MetroParks boundary exposure.

Focus:
- Benchmark NGROUP fixed effects against standard neighborhood fixed effects
  (e.g., census tract FE), using boundary-based MetroParks exposure.
- Use cluster-robust standard errors (cluster at tract GEOID).

Inputs:
- data_final/housing_regression_ready_with_tract.csv
- data_intermediate/v2/parcel_boundary_distances_v2.csv

Outputs:
- results/v2/housing_neighborhood_fe_benchmarks_v2.csv
- results/v2/housing_neighborhood_fe_benchmarks_v2_summary.txt
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


@dataclass(frozen=True)
class Spec:
    name: str
    formula: str
    fe_label: str


def fit_clustered(formula: str, data: pd.DataFrame, cluster_var: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    data_use = data.dropna(subset=[cluster_var]).copy()
    y, X = patsy.dmatrices(formula, data=data_use, return_type="dataframe")
    groups = data_use.loc[y.index, cluster_var]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    sales_path = os.path.join(project_root, "data_final", "housing_regression_ready_with_tract.csv")
    boundary_path = os.path.join(project_root, "data_intermediate", "v2", "parcel_boundary_distances_v2.csv")

    output_dir = os.path.join(project_root, "results", "v2")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "housing_neighborhood_fe_benchmarks_v2.csv")
    output_txt = os.path.join(output_dir, "housing_neighborhood_fe_benchmarks_v2_summary.txt")

    print("=" * 80)
    print("V2 HOUSING NEIGHBORHOOD FE BENCHMARKS (METRO BOUNDARY EXPOSURE)")
    print("=" * 80)

    df = pd.read_csv(sales_path, low_memory=False)
    df["PIN"] = df["PIN"].astype(str)
    boundary = pd.read_csv(boundary_path, low_memory=False)
    boundary["PIN"] = boundary["PIN"].astype(str)
    df = df.merge(boundary, on="PIN", how="left")

    df["ln_price"] = np.log(df["PRICE"])
    df["SALEDT"] = pd.to_datetime(df["SALEDT"])
    df["SaleYear"] = df["SALEDT"].dt.year

    required = ["GEOID", "NGROUP", "NBHD", "SCHOOLDIST", "NearPark_Boundary_Metro"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cluster at census tract (minimum defensible)
    cluster_var = "GEOID"

    specs: list[Spec] = [
        Spec(
            name="Metro_Binary_Bundled",
            fe_label="Year FE",
            formula="ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear)",
        ),
        Spec(
            name="Metro_Binary_SchoolDistFE",
            fe_label="Year FE + SchoolDist FE",
            formula="ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)",
        ),
        Spec(
            name="Metro_Binary_TractFE",
            fe_label="Year FE + Tract FE (GEOID)",
            formula="ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(GEOID)",
        ),
        Spec(
            name="Metro_Binary_NGROUP_FE",
            fe_label="Year FE + NGROUP FE",
            formula="ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(NGROUP)",
        ),
        Spec(
            name="Metro_Binary_NBHD_FE",
            fe_label="Year FE + NBHD FE",
            formula="ln_price ~ NearPark_Boundary_Metro + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(NBHD)",
        ),
    ]

    # Group size diagnostics for auditability
    group_stats: dict[str, dict[str, Any]] = {}
    for col in ["GEOID", "SCHOOLDIST", "NGROUP", "NBHD"]:
        vc = df[col].value_counts(dropna=True)
        group_stats[col] = {
            "n_groups": int(len(vc)),
            "mean_group_size": float(vc.mean()),
            "median_group_size": float(vc.median()),
            "min_group_size": int(vc.min()),
            "max_group_size": int(vc.max()),
        }

    rows: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    summary_lines.append("V2 NEIGHBORHOOD FE BENCHMARKS (CLUSTERED SE @ TRACT)")
    summary_lines.append("")
    summary_lines.append("Group diagnostics (count / mean / median / min / max):")
    for col, s in group_stats.items():
        summary_lines.append(
            f"  {col}: {s['n_groups']} / {s['mean_group_size']:.1f} / {s['median_group_size']:.1f} / {s['min_group_size']} / {s['max_group_size']}"
        )
    summary_lines.append("")

    for spec in specs:
        res = fit_clustered(spec.formula, df, cluster_var=cluster_var)
        b = float(res.params["NearPark_Boundary_Metro"])
        se = float(res.bse["NearPark_Boundary_Metro"])
        p = float(res.pvalues["NearPark_Boundary_Metro"])
        ci_low, ci_high = res.conf_int().loc["NearPark_Boundary_Metro"].tolist()

        pct = (np.exp(b) - 1.0) * 100.0
        pct_low = (np.exp(ci_low) - 1.0) * 100.0
        pct_high = (np.exp(ci_high) - 1.0) * 100.0

        rows.append(
            {
                "spec": spec.name,
                "fe": spec.fe_label,
                "cluster_var": cluster_var,
                "nearpark_coef": b,
                "nearpark_se": se,
                "nearpark_pvalue": p,
                "nearpark_ci_lower": float(ci_low),
                "nearpark_ci_upper": float(ci_high),
                "nearpark_pct_premium": pct,
                "nearpark_pct_premium_lower": pct_low,
                "nearpark_pct_premium_upper": pct_high,
                "n_obs": int(res.nobs),
                "r_squared": float(res.rsquared),
            }
        )

        summary_lines.append(
            f"{spec.name} ({spec.fe_label}): {pct:.2f}% "
            f"(95% CI {pct_low:.2f}%, {pct_high:.2f}%), p={p:.4g}, N={int(res.nobs)}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    with open(output_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nWrote: {output_csv}")
    print(f"Wrote: {output_txt}")


if __name__ == "__main__":
    main()
